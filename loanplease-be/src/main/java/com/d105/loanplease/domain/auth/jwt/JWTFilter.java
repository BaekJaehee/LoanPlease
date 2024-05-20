package com.d105.loanplease.domain.auth.jwt;

import io.jsonwebtoken.ExpiredJwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

@Slf4j
@RequiredArgsConstructor
@Component
public class JWTFilter extends OncePerRequestFilter {

    @Autowired
    private final TokenProvider tokenProvider;

    private final AntPathMatcher pathMatcher = new AntPathMatcher();
//    Ant 스타일 패턴 매칭 사용
//    Spring에서는 Ant 스타일의 경로 매칭을 지원하여 보다 유연하게 URL 패턴을 처리할 수 있습니다.
//    PathMatcher와 같은 클래스를 사용하여 Ant 스타일의 패턴을 사용할 수 있습니다.
    //허용 Uri를 관리하는 메서드
    private boolean isAllowedPath(String requestUri){
        List<String> allowedPaths = Arrays.asList("/api/server", "/api/upload", "/api/auth/nickname/**" ,"/swagger-ui/**","/swagger-resources/**",
                "/v3/api-docs/**","/api/auth/refresh","/api/auth/register","/signup");
        return allowedPaths.stream().anyMatch(p -> pathMatcher.match(p, requestUri));
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        try {
            if (isAllowedPath(request.getRequestURI())) {
                filterChain.doFilter(request, response);
                return;
            }
            log.info("DASF");
            String accessToken = tokenProvider.extractAccessToken(request).orElse(null);
            log.info("Access Token: " + accessToken);
            try {
                if (accessToken != null && tokenProvider.isTokenValid(accessToken)) {
                    Authentication authentication = tokenProvider.getAuthentication(accessToken);
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                    log.info("Access Token in ComeIn DoFilter ");
                    filterChain.doFilter(request, response);
                } else {
                    sendUnauthorizedResponse(response, "Access is Invalid or Expired");
                }
            } catch (ExpiredJwtException e) { //토큰이 만료 됐다면 401을 보낸다.
                log.info("Expired JWT token: {}", e.getMessage());
                sendUnauthorizedResponse(response, "401 Unauthorized - Token Expired");
            }
        } catch (Exception e) {
            SecurityContextHolder.clearContext();
            logger.error("ERROR : ", e);
            sendUnauthorizedResponse(response, e.getMessage());
        }
    }

    private void sendUnauthorizedResponse(HttpServletResponse response, String message) throws IOException {
        response.setContentType("application/json;charset=UTF-8");
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.getWriter().write("{\"error\": \"" + message + "\"}");
    }


}