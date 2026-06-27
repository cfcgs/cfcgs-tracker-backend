class AppError(Exception):
    """Base exception for application and domain errors."""


class UserNotFoundError(AppError):
    """Raised when a requested user does not exist."""

    def __init__(self) -> None:
        super().__init__("User not found.")


class UserAlreadyExistsError(AppError):
    """Raised when trying to create a duplicated user."""

    def __init__(
        self,
        message: str = "User with this username or email already exists",
    ) -> None:
        super().__init__(message)


class UsernameAlreadyExistsError(UserAlreadyExistsError):
    """Raised when trying to create a duplicated user username."""

    def __init__(self) -> None:
        super().__init__("User with this username already exists.")


class EmailAlreadyExistsError(UserAlreadyExistsError):
    """Raised when trying to create a duplicated user email."""

    def __init__(self) -> None:
        super().__init__("User with this email already exists.")


class PermissionDeniedError(AppError):
    """Raised when an action is not allowed for the current user."""

    def __init__(self) -> None:
        super().__init__("Not enough permissions")


class InvalidCredentialsError(AppError):
    """Raised when authentication credentials are invalid."""

    def __init__(self) -> None:
        super().__init__("Could not validate credentials.")


class InvalidLoginError(AppError):
    """Raised when login credentials are invalid."""

    def __init__(self) -> None:
        super().__init__("Incorrect username or password.")


class InvalidCurrentPasswordError(AppError):
    """Raised when the current password informed by the user is invalid."""

    def __init__(self) -> None:
        super().__init__("Current password is invalid.")


class PasswordConfirmationMismatchError(AppError):
    """Raised when the new password confirmation does not match."""

    def __init__(self) -> None:
        super().__init__("New password and confirmation do not match.")
