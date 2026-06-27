from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query

from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import User
from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_current_import_operator,
    get_import_service,
)
from src.cfcgs_tracker.entrypoints.api.schemas.common import FilterPage
from src.cfcgs_tracker.entrypoints.api.schemas.imports import (
    ImportJobList,
    ImportJobPublic,
)
from src.cfcgs_tracker.service_layer.services.imports import ImportService

router = APIRouter(prefix="/imports", tags=["imports"])
ImportOperator = Annotated[User, Depends(get_current_import_operator)]
ImportServiceDep = Annotated[ImportService, Depends(get_import_service)]


@router.post(
    "/", response_model=ImportJobPublic, status_code=HTTPStatus.CREATED
)
async def create_import(
    current_user: ImportOperator,
    service: ImportServiceDep,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="File name is required"
        )

    file_bytes = await file.read()

    try:
        import_job = await service.import_file(
            file_name=file.filename,
            file_bytes=file_bytes,
            current_user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)
        ) from exc

    return import_job


@router.get("/{import_job_id}", response_model=ImportJobPublic)
async def read_import_job(
    import_job_id: int,
    service: ImportServiceDep,
    current_user: ImportOperator,
):
    try:
        import_job = await service.get_import_job(import_job_id, current_user)
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc

    if not import_job:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Import job not found",
        )
    return import_job


@router.get("/", response_model=ImportJobList, status_code=HTTPStatus.OK)
async def read_import_jobs(
    service: ImportServiceDep,
    current_user: ImportOperator,
    filter_import_jobs: Annotated[FilterPage, Query()],
):
    import_jobs = await service.get_import_jobs(
        current_user=current_user,
        limit=filter_import_jobs.limit,
        offset=filter_import_jobs.offset,
    )
    return {"import_jobs": import_jobs}
