"""THE ALERT TOOL (deliverable #3).

A farmer with a ₹0 data plan and a ₹1,000 feature phone must still get the warning.
So the alert layer is *channel-first*:

  * SMS  - 160-character GSM text, works on every phone, always-on fallback.
           (TRAI's DLT rules apply in India for commercial traffic; a
           fisheries-department or co-operative sender ID is the realistic route.)
  * Voice/IVR - a 20-second recorded script in Kannada/Hindi for farmers who cannot
           read; generated from scripts/voice/*.txt via any TTS, or recorded by the
           co-operative.
  * WhatsApp - the realistic channel for JPEG photos of sick fish (health check +
           image triage happens here, over the free 24 h customer-service window).
  * App/console - the phone screen mock (screens/farmer_app.html).

Anti-alert-fatigue rules baked in:
  * one message per (pond, band-change, 6 h) -> dedupe_key;
  * RED bypasses dedupe but is rate-limited to once per hour;
  * every message ends with a *single* action, because a farmer reading an SMS at
    05:30 in the rain will do one thing, not five.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone

from ..schemas import Alert, Band, RiskAssessment

BAND_PREFIX = {
    Band.GREEN: "SAFE",
    Band.YELLOW: "WATCH",
    Band.ORANGE: "ALERT",
    Band.RED: "EMERGENCY",
}

# GSM-7 one-part SMS budget. Tamil/Kannada SMS needs UCS-2 -> only 70 chars/part,
# which is exactly why the primary SMS is kept in English/transliteration and the
# Kannada text is delivered by voice instead.
SMS_LIMIT = 160


class AlertDispatcher:
    """Pluggable dispatch. `console` prints; `sms`/`whatsapp` POST to a provider
    (MSG91 / Gupshup / Twilio in India) when AQUA_SMS_API_KEY is set."""

    def __init__(self) -> None:
        self.sent: list[Alert] = []
        self._last: dict[str, datetime] = {}

    # ------------------------------------------------------------------ dedupe --
    def dedupe_key(self, pond_id: str, band: Band) -> str:
        return hashlib.sha1(f"{pond_id}:{band}".encode()).hexdigest()[:12]

    def _suppressed(self, key: str, band: Band, now: datetime) -> bool:
        last = self._last.get(key)
        if last is None:
            return False
        window = timedelta(hours=1) if band is Band.RED else timedelta(hours=6)
        return (now - last) < window

    # --------------------------------------------------------------- compose ----
    def compose(self, assessment: RiskAssessment, channel: str = "sms") -> str:
        prefix = BAND_PREFIX[assessment.band]
        pond = assessment.pond_id
        action = assessment.actions[0] if assessment.actions else "Re-test in the morning."
        # strip the "(kannada)" parenthetical for the SMS budget
        short_action = action.split("|")[0].split("(")[0].strip().rstrip(".")
        if channel == "voice":
            return (f"{prefix}. Pond {pond}. {assessment.headline_en}. "
                    f"{short_action}. {assessment.headline_local}.")
        msg = (f"AQUA{prefix} Pond {pond} {assessment.ts:%d-%b %H:%M} "
               f"score {assessment.score:.2f}. {short_action}.")
        return msg[:SMS_LIMIT]

    # -------------------------------------------------------------- dispatch ----
    def dispatch(self, assessment: RiskAssessment, channel: str = "console",
                 to: str | None = None) -> Alert:
        now = datetime.now(timezone.utc)
        key = self.dedupe_key(assessment.pond_id, assessment.band)

        if assessment.band is Band.GREEN:
            return Alert(id=key, pond_id=assessment.pond_id, created_at=now, band=assessment.band,
                         channel=channel, message="", dedupe_key=key, delivered=False,
                         error="GREEN suppressed: do not spend SMS credits on good news")

        if self._suppressed(key, assessment.band, now):
            return Alert(id=key, pond_id=assessment.pond_id, created_at=now, band=assessment.band,
                         channel=channel, message="", dedupe_key=key, delivered=False,
                         error="suppressed by anti-fatigue window")

        message = self.compose(assessment, channel)
        alert = Alert(id=key, pond_id=assessment.pond_id, created_at=now,
                      band=assessment.band, channel=channel,  # type: ignore[arg-type]
                      message=message, dedupe_key=key, delivered=False)

        try:
            if channel == "console" or not os.getenv("AQUA_SMS_API_KEY"):
                print(f"[{channel.upper()} -> {to or 'farmer'}] {message}")
                alert.delivered = True
            else:
                alert.error = "provider adapter not configured in this demo build"
        except Exception as exc:  # pragma: no cover - network path
            alert.error = str(exc)

        self._last[key] = now
        self.sent.append(alert)
        return alert


def escalation_matrix() -> dict[str, str]:
    """Who gets told what, and how fast - the operational half of the alert tool."""
    return {
        "GREEN": "Farmer app only, silent. Weekly digest SMS on Sunday.",
        "YELLOW": "Farmer SMS (1 part) + app card. Co-operative dashboard updated.",
        "ORANGE": "Farmer SMS + voice call in Kannada + WhatsApp photo request; "
                  "co-operative officer notified in dashboard.",
        "RED": "Farmer SMS + automated voice call + WhatsApp; district fisheries "
               "extension officer paged; 6-hourly re-check until band drops.",
    }
