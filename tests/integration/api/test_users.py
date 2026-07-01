from http import HTTPStatus

from src.cfcgs_tracker.domain.models import UserRole
from src.cfcgs_tracker.entrypoints.api.schemas.users import (
    UserCreate,
    UserPublic,
    UserUpdate,
)


def test_create_user_as_admin(client, admin_token):
    response = client.post(
        "/users",
        json=UserCreate(
            username="alice",
            email="alice@example.com",
            password="secret",
            role=UserRole.importer,
        ).model_dump(mode="json"),
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )

    assert response.status_code == HTTPStatus.CREATED
    assert response.json() == {
        "username": "alice",
        "email": "alice@example.com",
        "id": 2,
        "role": "importer",
        "is_active": True,
    }


def test_create_user_requires_admin(client, token):
    response = client.post(
        "/users",
        json=UserCreate(
            username="alice",
            email="alice@example.com",
            password="secret",
            role=UserRole.importer,
        ).model_dump(mode="json"),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "Not enough permissions"}


def test_create_user_with_same_username(client, admin_token, user):
    response = client.post(
        "/users",
        json={
            "username": user.username,
            "email": "email@example.com",
            "password": "password",
            "role": "importer",
        },
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": "User with this username already exists."
    }


def test_create_user_with_same_email(client, admin_token, user):
    response = client.post(
        "/users",
        json={
            "email": user.email,
            "username": "NovoUser",
            "password": "password",
            "role": "importer",
        },
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": "User with this email already exists."
    }


def test_read_users_as_admin(client, user, admin_user, admin_token):
    user_schema = UserPublic.model_validate(user).model_dump(mode="json")
    admin_schema = UserPublic.model_validate(admin_user).model_dump(
        mode="json"
    )
    response = client.get(
        "/users",
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"users": [user_schema, admin_schema]}


def test_read_users_requires_admin(client, token):
    response = client.get(
        "/users",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "Not enough permissions"}


def test_read_user_as_importer_reads_self(client, user, token):
    response = client.get(
        f"/users/{user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == UserPublic.model_validate(user).model_dump(
        mode="json"
    )


def test_read_user_as_admin_reads_any_user(client, user, admin_token):
    response = client.get(
        f"/users/{user.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == UserPublic.model_validate(user).model_dump(
        mode="json"
    )


def test_read_user_requires_own_user_for_importer(
    client,
    other_user,
    token,
):
    response = client.get(
        f"/users/{other_user.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "Not enough permissions"}


def test_read_user_not_found_for_admin(client, admin_token):
    response = client.get(
        "/users/999",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json().get("detail") == "User not found."

def test_update_user_as_importer_on_self(client, user, token):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username="alice",
            email="alice@example.com.br",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "username": "alice",
        "email": "alice@example.com.br",
        "id": 1,
        "role": "importer",
        "is_active": True,
    }


def test_update_user_forbidden_for_admin_on_other_user(
    client, user, admin_token
):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username="alice",
            email="alice@example.com.br",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json().get("detail") == "Not enough permissions"


def test_update_user_as_admin_on_self(client, admin_user, admin_token):
    response = client.put(
        f"/users/{admin_user.id}",
        json=UserUpdate(
            username="admin-updated",
            email="admin-updated@example.com",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "username": "admin-updated",
        "email": "admin-updated@example.com",
        "id": admin_user.id,
        "role": "admin",
        "is_active": True,
    }


def test_update_not_enough_permissions(client, other_user, token):
    response = client.put(
        f"/users/{other_user.id}",
        json=UserUpdate(
            username="alice",
            email="ali@example.com.br",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json().get("detail") == "Not enough permissions"


def test_update_email_already_exists_error(client, user, other_user, token):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username="alice",
            email=other_user.email,
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": "User with this email already exists.",
    }


def test_update_username_already_exists_error(
    client,
    user,
    other_user,
    token,
):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username=other_user.username,
            email="joaquim@example.com",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {
        "detail": "User with this username already exists.",
    }


def test_update_user_password_with_current_password(client, user, token):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username=user.username,
            email=user.email,
            current_password="testtest",
            new_password="new-secret",
            new_password_confirmation="new-secret",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == user.id


def test_update_user_password_rejects_invalid_current_password(
    client,
    user,
    token,
):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username=user.username,
            email=user.email,
            current_password="wrong-password",
            new_password="new-secret",
            new_password_confirmation="new-secret",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json() == {"detail": "Current password is invalid."}


def test_update_user_password_requires_matching_confirmation(
    client,
    user,
    token,
):
    response = client.put(
        f"/users/{user.id}",
        json=UserUpdate(
            username=user.username,
            email=user.email,
            current_password="testtest",
            new_password="new-secret",
            new_password_confirmation="different-secret",
        ).model_dump(),
        headers={
            "Authorization": f"Bearer {token}",
        },
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {
        "detail": "New password and confirmation do not match."
    }


def test_delete_user_as_importer_on_self(client, user, token):
    response = client.delete(
        f"/users/{user.id}",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "User deactivated"}


def test_delete_user_as_admin_on_any_user(client, user, admin_token):
    response = client.delete(
        f"/users/{user.id}",
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"message": "User deactivated"}


def test_update_user_role_as_admin(client, user, admin_token):
    response = client.patch(
        f"/users/{user.id}/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["id"] == user.id
    assert response.json()["role"] == "admin"


def test_update_user_role_requires_admin(client, user, token):
    response = client.patch(
        f"/users/{user.id}/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json() == {"detail": "Not enough permissions"}


def test_delete_user_not_enough_permissions(client, other_user, token):
    response = client.delete(
        f"/users/{other_user.id}",
        headers={
            "Authorization": f"Bearer {token}",
        },
    )
    assert response.status_code == HTTPStatus.FORBIDDEN
    assert response.json().get("detail") == "Not enough permissions"


def test_delete_user_as_admin_on_unexisting_user_id(client, admin_token):
    response = client.delete(
        "/users/100",
        headers={
            "Authorization": f"Bearer {admin_token}",
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json().get("detail") == "User not found."
