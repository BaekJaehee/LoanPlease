package com.d105.loanplease.global.util;

import jakarta.servlet.http.Cookie;

public class CookieUtils {

    public static Cookie createCookie(String key, String value, int maxAge, boolean isHttpOnly) {
        Cookie cookie = new Cookie(key, value);
        cookie.setMaxAge(maxAge);  // 유효 기간 설정
        cookie.setPath("/");
        cookie.setHttpOnly(isHttpOnly);  // HttpOnly 설정
        return cookie;
    }

    public static Cookie createHttpOnlyCookie(String key, String value, int maxAge) {
        return createCookie(key, value, maxAge, true);
    }





}
