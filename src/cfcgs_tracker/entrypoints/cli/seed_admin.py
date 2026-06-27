import asyncio

from src.cfcgs_tracker.adapters.orm import session_factory
from src.cfcgs_tracker.domain.models import User, UserRole
from src.cfcgs_tracker.service_layer.security import get_password_hash
from src.cfcgs_tracker.settings import Settings


async def seed_initial_admin() -> None:
    settings = Settings()

    if not (
        settings.INITIAL_ADMIN_USERNAME
        and settings.INITIAL_ADMIN_EMAIL
        and settings.INITIAL_ADMIN_PASSWORD
    ):
        print(
            "[seed-admin] skipped: INITIAL_ADMIN_USERNAME, "
            "INITIAL_ADMIN_EMAIL and INITIAL_ADMIN_PASSWORD were not fully "
            "configured."
        )
        return

    async with session_factory() as session:
        from sqlalchemy import select

        existing_user = await session.scalar(
            select(User).where(User.email == settings.INITIAL_ADMIN_EMAIL)
        )

        if existing_user:
            changed = False

            if existing_user.role != UserRole.admin:
                existing_user.role = UserRole.admin
                changed = True

            if existing_user.username != settings.INITIAL_ADMIN_USERNAME:
                existing_user.username = settings.INITIAL_ADMIN_USERNAME
                changed = True

            if changed:
                await session.commit()
                print(
                    "[seed-admin] existing user updated to admin role: "
                    f"{existing_user.email}"
                )
            else:
                print(
                    "[seed-admin] existing admin already present: "
                    f"{existing_user.email}"
                )
            return

        existing_username = await session.scalar(
            select(User).where(
                User.username == settings.INITIAL_ADMIN_USERNAME
            )
        )
        if existing_username:
            print(
                "[seed-admin] skipped: username already exists with another "
                f"email: {settings.INITIAL_ADMIN_USERNAME}"
            )
            return

        admin_user = User(
            username=settings.INITIAL_ADMIN_USERNAME,
            email=settings.INITIAL_ADMIN_EMAIL,
            password=get_password_hash(settings.INITIAL_ADMIN_PASSWORD),
            role=UserRole.admin,
        )
        session.add(admin_user)
        await session.commit()

        print(
            "[seed-admin] initial admin created successfully: "
            f"{admin_user.email}"
        )


def main() -> None:
    asyncio.run(seed_initial_admin())


if __name__ == "__main__":
    main()
