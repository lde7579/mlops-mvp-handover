"""오류 응답: 모든 오류를 {"error": "이름", "message": "설명"} 형식으로 통일한다 (docs/api-v1.md 1장).

- error 는 HTTP 상태와 1:1인 문자열 이름이다 (숫자가 아님).
- FastAPI 기본 입력 검증 오류(422)는 400 bad_request 로 바꾼다 (문서 코드표에 422 가 없다).
- 401 에는 WWW-Authenticate: Bearer, 405 에는 Allow 헤더가 붙는다 (HTTP 표준).
- 처리되지 않은 예외는 500 server_error 이고, 내부 정보(트레이스백 등)는 응답에 싣지 않는다.
"""
import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api_models import ErrorResponse

log = logging.getLogger("backend")

ERROR_NAMES = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
    405: "method_not_allowed", 409: "conflict", 500: "server_error", 503: "service_unavailable",
}
DEFAULT_MESSAGES = {
    400: "요청 형식이 올바르지 않습니다", 401: "인증이 필요합니다", 403: "이 작업을 할 권한이 없습니다",
    404: "찾을 수 없는 주소입니다", 405: "이 주소에서 허용되지 않는 메서드입니다", 409: "현재 상태와 충돌합니다",
    500: "서버 내부 오류가 발생했습니다", 503: "지금은 요청을 처리할 수 없습니다",
}
BEARER_HEADER = {"WWW-Authenticate": "Bearer"}


class ApiError(Exception):
    """엔드포인트에서 던지는 오류. 상태 코드와 한국어 메시지를 담는다."""

    def __init__(self, status: int, message: Optional[str] = None, headers: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.message = message or DEFAULT_MESSAGES.get(status, "오류가 발생했습니다")
        self.headers = headers


def error_name(status: int) -> str:
    if status in ERROR_NAMES:
        return ERROR_NAMES[status]
    return "server_error" if status >= 500 else "bad_request"


def error_response(status: int, message: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse({"error": error_name(status), "message": message}, status_code=status, headers=headers)


_VALIDATION_TEXT = {
    "missing": "필수 항목입니다", "json_invalid": "올바른 JSON 이 아닙니다", "string_type": "문자열이어야 합니다",
    "string_too_short": "비어 있을 수 없습니다", "string_too_long": "너무 깁니다",
    "extra_forbidden": "허용되지 않는 항목입니다", "enum": "허용된 값이 아닙니다", "literal_error": "허용된 값이 아닙니다", "int_parsing": "정수여야 합니다",
    "int_type": "정수여야 합니다", "greater_than_equal": "허용 범위보다 작습니다", "less_than_equal": "허용 범위보다 큽니다",
    "model_attributes_type": "JSON 객체여야 합니다",
}


def validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = [str(x) for x in err.get("loc", ()) if x not in ("query", "path", "header", "cookie")]
        if loc == ["body"]:  # 본문 전체가 없거나 JSON 이 아님
            parts.append("요청 본문이 필요합니다" if err.get("type") == "missing"
                         else _VALIDATION_TEXT.get(err.get("type", ""), "요청 본문이 올바르지 않습니다"))
            continue
        field = ".".join(x for x in loc if x != "body")
        text = _VALIDATION_TEXT.get(err.get("type", ""), "올바르지 않습니다")
        parts.append(f"{field}: {text}" if field else text)
    return "; ".join(parts) or DEFAULT_MESSAGES[400]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)  # 없는 주소(404), 허용되지 않는 메서드(405) 등
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, DEFAULT_MESSAGES.get(exc.status_code, "요청을 처리할 수 없습니다"),
                              dict(exc.headers) if exc.headers else None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return error_response(400, validation_message(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("처리되지 않은 오류", exc_info=exc)
        return error_response(500, DEFAULT_MESSAGES[500])


def install_openapi(app: FastAPI) -> None:
    """자동 생성되는 /openapi.json 에서 422 응답 설명을 400(ErrorResponse)으로 바꾼다."""
    original = app.openapi

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = original()
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        schemas.setdefault("ErrorResponse", ErrorResponse.model_json_schema())
        for path_item in schema.get("paths", {}).values():
            for operation in path_item.values():
                responses = operation.get("responses", {})
                if "422" in responses:
                    del responses["422"]
                    responses["400"] = {
                        "description": "잘못된 요청 (입력 검증 실패)",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
                    }
        for name in ("HTTPValidationError", "ValidationError"):
            schemas.pop(name, None)
        return schema

    app.openapi = custom_openapi
