class ReleaseIntegrityError(RuntimeError):
    """A release artifact failed a cryptographic or structural integrity check."""


class MigrationError(RuntimeError):
    """Base error for forward-only migration refusal."""


class MigrationChecksumError(MigrationError):
    """An already-applied migration no longer matches its recorded checksum."""


class MigrationHistoryError(MigrationError):
    """The database migration history is not a prefix of the supplied plan."""

