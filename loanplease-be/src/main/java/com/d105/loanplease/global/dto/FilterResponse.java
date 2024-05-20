package com.d105.loanplease.global.dto;

import com.d105.loanplease.global.exception.ErrorCode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.HashMap;

@Component
public class FilterResponse {

    private final ObjectMapper mapper = new ObjectMapper();

    // Filter 정상 종료에 따른 Response 설정
    public <T> HttpServletResponse ok(HttpServletResponse response, T data)
            throws IOException {

        HashMap<String, Object> responseBody = new HashMap<>();
        responseBody.put("status", "success");
        responseBody.put("message", "성공하였습니다.");
        responseBody.put("data", data);

        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().write(mapper.writeValueAsString(responseBody));

        return response;
    }

    public void error(HttpServletResponse response, ErrorCode errorCode) throws IOException {

        HashMap<String, Object> responseBody = new HashMap<>();
        responseBody.put("status", "fail");
        responseBody.put("message", errorCode.getMessage());
        responseBody.put("data", null);

        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");
        response.getWriter().write(mapper.writeValueAsString(responseBody));
        response.setStatus(errorCode.getHttpStatus().value());
    }

    public void error(HttpServletResponse response, String msg) throws IOException {
        response.sendError(HttpServletResponse.SC_UNAUTHORIZED, msg);
    }

}