package com.d105.loanplease.domain.auth.jwt;

import com.d105.loanplease.domain.auth.repository.TokenRepository;
import com.d105.loanplease.domain.user.repository.UserRepository;
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
    @Autowired
    private final TokenRepository tokenRepository;

    private final AntPathMatcher pathMatcher = new AntPathMatcher();
//    Ant 스타일 패턴 매칭 사용
//    Spring에서는 Ant 스타일의 경로 매칭을 지원하여 보다 유연하게 URL 패턴을 처리할 수 있습니다.
//    PathMatcher와 같은 클래스를 사용하여 Ant 스타일의 패턴을 사용할 수 있습니다.
    //허용 Uri를 관리하는 메서드
    private boolean isAllowedPath(String requestUri){
        List<String> allowedPaths = Arrays.asList("/api/friends","/api/server", "/api/upload", "/api/auth/nickname/**" ,"/swagger-ui/**","/swagger-resources/**",
                "/v3/api-docs/**","/api/refresh","/api/auth/register","/signup");
//        return allowedPaths.stream().anyMatch(requestUri::startsWith);
        //,"/swagger-ui/index.html","/swagger-ui/index.html" //
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
//    @Override
//    protected void doFilterInternal(
//            HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
//            throws ServletException, IOException {
//        try {
//            if (isAllowedPath(request.getRequestURI())) {
//                filterChain.doFilter(request, response);
//                return;
//            }
//
//            String accessToken = tokenProvider.extractAccessToken(request).orElse(null);
//            String refreshToken = tokenProvider.extractRefreshToken(request).orElse(null);
//
//            log.info("accessToken : "+accessToken);
//            log.info("refreshToken : "+ refreshToken);
//            if (accessToken != null && tokenProvider.isTokenValid(accessToken)) {
//                Authentication authentication = tokenProvider.getAuthentication(accessToken);
//                SecurityContextHolder.getContext().setAuthentication(authentication);
//                filterChain.doFilter(request, response);
//            } else if (refreshToken != null && tokenProvider.isTokenValid(refreshToken)) {
//                // 리프레시 토큰이 유효한 경우 새 엑세스 토큰 발급
//                String newAccessToken = tokenProvider.createAccessJwt(tokenProvider.getAuthentication(refreshToken));
//                response.setHeader("Authorization", "Bearer " + newAccessToken);
//                logger.info("New access token issued.");
//                Authentication authentication = tokenProvider.getAuthentication(newAccessToken);
//                SecurityContextHolder.getContext().setAuthentication(authentication);
//                filterChain.doFilter(request, response);
//            } else {
//                sendUnauthorizedResponse(response, "Access is Invalid or Expired");
//            }
//        } catch (Exception e) {
//            SecurityContextHolder.clearContext();
//            logger.error("Authentication ERROR : ", e);
//            sendUnauthorizedResponse(response, "Authentication error: " + e.getMessage());
//        }
//    }

//    @Override
//    protected void doFilterInternal(
//            HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
//            throws ServletException, IOException {
//        try{
//            if (isAllowedPath(request.getRequestURI())){ //허용된 URI인지 확인한다.
//                // 특정 경로에 대해 필터링 없이 진행한다.
//                filterChain.doFilter(request,response);
//                return;
//            }
////            //토큰
////            String token = authorization;
//            String accessToken = tokenProvider.extractAccessToken(request).orElse(null);
//            String refreshToken = tokenProvider.extractRefreshToken(request).orElse(null);
//
//            logger.info(accessToken+":"+refreshToken);
//
//            if (tokenRepository.findByAccessToken(accessToken).isEmpty()){
//                sendUnauthorizedResponse(response,"Access is Invalid");
//                logger.info("Access is Invalid");
//                return;
//            }
//            //
//            //filterChain.doFilter(request,response);
//            //
//            if (accessToken!=null && tokenProvider.isTokenValid(accessToken)){
//                Authentication authentication = tokenProvider.getAuthentication(accessToken);
//                SecurityContextHolder.getContext().setAuthentication(authentication);
//                logger.info("잘 드왔도다 ");
//                logger.info("잘 드왔도다 ");logger.info("잘 드왔도다 ");logger.info("잘 드왔도다 ");logger.info("잘 드왔도다 ");
//                filterChain.doFilter(request,response);
//            }
//        }catch (ExpiredJwtException e){
//            //401에러
//
//            sendUnauthorizedResponse(response,"401");
//        }
//        catch (Exception e){
//            SecurityContextHolder.clearContext();
//            logger.error("Authentication ERROR : ", e);
//            sendUnauthorizedResponse(response, "Authentication  error :" + e.getMessage());
//        }
//    }


    //인가되지 않은 사용자에게 띄어줄 페이지
//    private void sendUnauthorizedResponse(HttpServletResponse response, String message) throws IOException {
//        response.setContentType("application/json;charset=UTF-8");
//        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
//        response.getWriter().write(message);
//    }

    private void sendUnauthorizedResponse(HttpServletResponse response, String message) throws IOException {
        response.setContentType("application/json;charset=UTF-8");
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.getWriter().write("{\"error\": \"" + message + "\"}");
    }


}