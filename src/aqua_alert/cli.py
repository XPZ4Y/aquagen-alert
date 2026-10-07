"""CLI - the demo path a judge/officer can run in one line.

    uv run aqua-alert catalog
    uv run aqua-alert demo
    uv run aqua-alert forecast --lat 12.97 --lon 77.59
    uv run aqua-alert triage photo.jpg
    uv run aqua-alert serve            # farmer screen + API at :8000
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import pipeline, risk
from .alerts.dispatcher import AlertDispatcher, escalation_matrix
from .disease import classifier
from .schemas import Band, PondReading
from .sources import registry, synthetic, weather

app = typer.Typer(add_completion=False, help="Early disease & water-quality alerts for small farms")
console = Console()

COLOUR = {Band.GREEN: "green", Band.YELLOW: "yellow", Band.ORANGE: "dark_orange", Band.RED: "red"}


@app.command()
def catalog(kind: str | None = typer.Option(None, help="filter: timeseries|images|weather|...")) -> None:
    """Print the public-data catalogue (deliverable #2)."""
    t = Table(title="Public data replacing physical sensors", show_lines=True)
    for col in ("id", "kind", "replaces sensor", "license"):
        t.add_column(col)
    for e in registry.CATALOG:
        if kind and e.kind.value != kind:
            continue
        t.add_row(e.id, e.kind.value, e.replaces_sensor, e.license)
    console.print(t)
    console.print(f"[dim]{len(registry.CATALOG)} entries - full detail: GET /catalog[/dim]")


@app.command()
def demo(seed: int = 7, channel: str = "console") -> None:
    """Offline end-to-end demo on the bundled sample pond log."""
    d = pipeline.demo_day(seed=seed, dispatcher=AlertDispatcher(), channel=channel)
    _print_assessment(d.assessment)
    if d.alerts and d.alerts[0].delivered:
        console.print(Panel(d.alerts[0].message, title="SMS sent to farmer", style="bold red"))


@app.command()
def assess(do: float = typer.Option(..., "--do", help="dissolved oxygen mg/L"),
           ph: float = typer.Option(..., help="pH"),
           temp: float = typer.Option(..., help="water temp C"),
           tan: float = typer.Option(None, help="total ammonia mg/L (test strip)"),
           turbidity: float = typer.Option(None, help="NTU (visual/disc estimate)"),
           species: str = "rohu", pond: str = "KA-BLR-001") -> None:
    """Assess one morning reading typed in by the farmer."""
    r = PondReading(pond_id=pond, do_mgl=do, ph=ph, water_temp_c=temp,
                    ammonia_total_mgl=tan, turbidity_ntu=turbidity, source="manual kit")
    a = risk.assess(r, species=species)
    _print_assessment(a)


@app.command()
def forecast(lat: float = 12.9716, lon: float = 77.5946, days: int = 3) -> None:
    """Next nights' oxygen-crash risk from free Open-Meteo data (no DO probe)."""
    try:
        rows = weather.fetch_daily(lat, lon, date.today(), date.today() + timedelta(days=days - 1))
    except Exception as exc:
        console.print(f"[red]weather fetch failed:[/red] {exc}\n"
                      "Offline? The rest of the system still works.")
        raise typer.Exit(1) from exc
    t = Table(title=f"Night oxygen outlook - {lat},{lon}")
    for col in ("date", "band", "why", "virtual water temp C", "rain mm"):
        t.add_column(col)
    for d in rows:
        band, reason = risk.do_crash_forecast(d.t_max, d.t_min, d.wind_ms, d.precip_mm,
                                              d.cloud_pct, None)
        t.add_row(d.day.isoformat(), f"[{COLOUR[band]}]{band.value}[/]", reason,
                  str(d.virtual_water_temp()), f"{d.precip_mm:.1f}")
    console.print(t)
    console.print("[dim]Weather data by Open-Meteo.com (CC BY 4.0)[/dim]")


@app.command()
def triage(image: Path, species: str = "rohu") -> None:
    """Photo -> disease triage (uses the public datasets in the catalogue)."""
    if not image.exists():
        console.print(f"[red]no such file:[/red] {image}")
        raise typer.Exit(1)
    p = classifier.predict(image, species=species)
    console.print(Panel(
        f"[bold]{p.label}[/bold]  confidence {p.confidence:.2f}  ({p.backend})\n"
        f"triage: [bold]{p.triage}[/bold]\n\n{p.advice_en}\n\n{p.advice_local}\n\n"
        f"top-3: {', '.join(f'{k} {v:.2f}' for k, v in p.top_k)}",
        title="Disease triage"))


@app.command()
def sample_data(days: int = 14) -> None:
    """(Re)generate the bundled offline sample pond log."""
    p = synthetic.write_csv(days=days)
    console.print(f"wrote [bold]{p}[/bold] ({len(synthetic.load_csv())} rows)")


@app.command()
def matrix() -> None:
    """Who gets told what, and how fast."""
    for band, rule in escalation_matrix().items():
        console.print(Panel(rule, title=band))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API + farmer alert screen."""
    import uvicorn
    uvicorn.run("aqua_alert.api.server:app", host=host, port=port, reload=reload)


@app.command()
def replay(jsonl: Path, species: str = "rohu", channel: str = "console") -> None:
    """Replay an exported reading log (JSONL of PondReading) to see the alert stream."""
    disp = AlertDispatcher()
    n = 0
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = PondReading(**json.loads(line))
        pipeline.assess_pond(r, species=species, dispatcher=disp, channel=channel,
                             include_forecast=False)
        n += 1
    console.print(f"replayed {n} readings; {len(disp.sent)} alerts dispatched")


def _print_assessment(a) -> None:
    style = COLOUR[a.band]
    body = [f"[bold {style}]{a.band.value} - {a.headline_en}[/bold {style}]",
            f"[dim]{a.headline_local} | score {a.score:.2f} | {a.species}[/dim]", ""]
    body += [f"{i + 1}. {x}" for i, x in enumerate(a.actions)]
    if a.drivers:
        body.append("")
        table = Table(show_header=True, box=None)
        for c in ("parameter", "value", "status", "severity"):
            table.add_column(c)
        for r in a.drivers:
            table.add_row(r.parameter, f"{r.value:.3g} {r.unit}", r.status, f"{r.severity:.2f}")
        console.print(Panel("\n".join(body), title=f"Pond {a.pond_id}", style=style))
        console.print(table)
    else:
        console.print(Panel("\n".join(body), title=f"Pond {a.pond_id}", style=style))
    if a.missing:
        console.print(f"[dim]missing parameters: {', '.join(a.missing)}[/dim]")
    for n in a.notes:
        console.print(f"[dim]note: {n}[/dim]")


if __name__ == "__main__":
    app()
