package com.d105.loanplease.global.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
public enum ErrorCode {
    // 회원
    MEMBER_NOT_EXIST(HttpStatus.BAD_REQUEST, "M-001", "존재하지 않는 회원입니다."),
    EMAIL_EXIST(HttpStatus.BAD_REQUEST, "M-002", "이미 존재하는 이메일입니다."),


    EMPTY_TOKEN(HttpStatus.UNAUTHORIZED,"P-007", "토큰을 보내지 않았습니다."),
    ANOTHER_AUTH_ERROR(HttpStatus.UNAUTHORIZED,"P-006", "토큰 외의 Auth 에러입니다."),
    EXPIRED_TOKEN(HttpStatus.UNAUTHORIZED,"P-001", "만료된 토큰 입니다."),
    INVALID_TOKEN(HttpStatus.UNAUTHORIZED,"P-002", "유효하지 않은 토큰 입니다."),
    CANT_FIND_REFRESH(HttpStatus.UNAUTHORIZED,"P-004", "해당 토큰을 찾을 수 없습니다."),
    ALREADY_LOGOUT(HttpStatus.UNAUTHORIZED,"P-005", "이미 로그아웃한 토큰 입니다."),
    PERMISSION_DENIED(HttpStatus.BAD_REQUEST,"P-003","권한에러 "),


    // Validation
    NOT_VALID_REQUEST(HttpStatus.BAD_REQUEST, "I-001", "요청변수가 유효하지 않습니다.");

    private HttpStatus httpStatus;
    private String errorCode;
    private String message;

    ErrorCode(HttpStatus httpStatus, String errorCode, String message) {
        this.httpStatus = httpStatus;
        this.errorCode = errorCode;
        this.message = message;
    }
}
