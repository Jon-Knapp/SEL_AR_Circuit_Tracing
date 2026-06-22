# labjack_interface.py
#
# A small, readable wrapper around the LabJack U12's digital-input read, used
# as the CONTINUITY SENSOR for the Continuity Annotation System.
#
# Plain-language summary:
#   The two test probes are wired to the LabJack U12 (NOT to the circuit
#   terminals directly). When the probes complete a circuit, the U12's digital
#   input reads one value (by default state == 1); when the circuit is open it
#   reads the other (state == 0). This module turns that single yes/no
#   electrical reading into a simple True / False / None in Python:
#
#       True  -> continuity (the two probe tips are electrically connected)
#       False -> open       (no continuity)
#       None  -> the LabJack is unavailable (not plugged in / not responding)
#
#   The CAMERA, not the LabJack, decides WHICH terminals are involved. This
#   module only answers "are the two probe tips electrically connected right
#   now?" main.py combines that answer with the terminal the camera sees each
#   probe resting on, and records the pair.
#
# Design note: this wrapper takes its settings as plain function arguments
# (which channel, which state means "connected"); it does NOT import config.
# That mirrors object_detection.py, so the hardware code stays self-contained
# and is easy to reuse or test on its own.
#
# Import safety: if the 'u12' library cannot be imported (for example on a
# machine without LabJack's driver installed), this module still imports
# cleanly and simply reports the device as unavailable, so the rest of the
# program can run.

# Try to import LabJack's u12 library. If it is missing we remember why and
# carry on; the device will just report as unavailable.
try:
    import u12
    U12_IMPORT_ERROR = None
except Exception as error:            # pragma: no cover - depends on the machine
    u12 = None
    U12_IMPORT_ERROR = str(error)


class LabJackContinuity:
    """Talks to one digital input on a LabJack U12 and reports continuity."""

    def __init__(self, channel, continuity_state):
        """
        channel          : which U12 digital input to read (0, 1, 2, ...). This
                           matches the channel used in test_labjack_u12.py.
        continuity_state : the raw state value (0 or 1) that means "the probes
                           are connected". test_labjack_u12.py treats state == 1
                           as connected, so that is what config passes in. Flip
                           it to 0 if your wiring reads the opposite way.
        """
        self.channel = channel
        self.continuity_state = continuity_state
        self.device = None
        self.connected = False
        self.last_error = None
        self.try_open()

    def try_open(self):
        """
        Attempt to (re)open the LabJack and confirm it responds with a test
        read. Safe to call repeatedly. Returns True if the device is now
        connected, False otherwise.

        Honest caveat: opening - and especially RE-opening after an unplug -
        could not be tested here without the physical device. If the LabJack is
        unplugged and replugged mid-session and the status does not recover,
        restart the program; that is the guaranteed-clean path.
        """
        if u12 is None:
            self.connected = False
            self.last_error = U12_IMPORT_ERROR
            return False
        try:
            self.device = u12.U12()
            # A test read: if this succeeds, the device is really present.
            self.device.eDigitalIn(self.channel)
            self.connected = True
            self.last_error = None
        except Exception as error:
            self.device = None
            self.connected = False
            self.last_error = str(error)
        return self.connected

    def read_continuity(self):
        """
        Return True (continuity), False (open), or None (device unavailable).

        If a read throws - which is what happens if the LabJack is unplugged
        mid-session - we mark the device disconnected and return None, so the
        caller can show the "not connected" warning and stop recording
        connections until it comes back.
        """
        if not self.connected or self.device is None:
            return None
        try:
            result = self.device.eDigitalIn(self.channel)
            state = result["state"]
            return state == self.continuity_state
        except Exception as error:
            self.device = None
            self.connected = False
            self.last_error = str(error)
            return None