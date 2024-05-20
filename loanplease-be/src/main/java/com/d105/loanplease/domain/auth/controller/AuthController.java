package com.d105.loanplease.domain.auth.controller;


import com.d105.loanplease.domain.auth.dto.TokenResDto;
import com.d105.loanplease.domain.auth.entity.Token;
import com.d105.loanplease.domain.auth.jwt.TokenProvider;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.exception.ErrorCode;
import com.d105.loanplease.global.service.RedisService;
import com.d105.loanplease.global.util.BaseResponseBody;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;


@RestController
@RequiredArgsConstructor
@RequestMapping("/api")
public class AuthController {
    @Value("${spring.jwt.access.header}")
    private String accessHeader;
    @Value("${spring.jwt.refresh.header}")
    private String refreshHeader;
    private final TokenProvider tokenProvider;
    private final HttpServletRequest request;
    private final RedisService redisService;
    private final UserRepository userRepository;
    @GetMapping("/auth/refresh")
    public ResponseEntity<? extends BaseResponseBody> regenerateAccessToken() {
        String headerValue = request.getHeader("Authorization");
        if (headerValue != null && headerValue.startsWith("Bearer ")) {
            String token = headerValue.substring(7);
            String redisToken = redisService.getValues(token);
            Optional<User> user = userRepository.findByEmail(redisToken);
            if (user.isPresent()){
                String email = user.orElseThrow().getEmail();
                String accessToken = tokenProvider.createAccessJwt(email);
                String refreshToken = tokenProvider.createRefreshJwt(accessToken);
                TokenResDto tokenResDto = TokenResDto.builder().accessToken(accessToken).refreshToken(refreshToken).build();
                return ResponseEntity.status(HttpStatus.CREATED).body(BaseResponseBody.of("200", tokenResDto));
            }
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(BaseResponseBody.error(ErrorCode.NOT_VALID_REQUEST.getErrorCode(),ErrorCode.NOT_VALID_REQUEST.getMessage()));
            // 토큰 처리 로직
        } else {
            // 적절한 에러 처리
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(BaseResponseBody.error(ErrorCode.NOT_VALID_REQUEST.getErrorCode(),ErrorCode.NOT_VALID_REQUEST.getMessage()));
        }
    }
}
