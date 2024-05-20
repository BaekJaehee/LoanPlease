package com.d105.loanplease.global.config;
import com.d105.loanplease.domain.auth.jwt.JWTFilter;
import com.d105.loanplease.domain.auth.jwt.TokenProvider;
import com.d105.loanplease.domain.auth.oauth.CustomSuccessHandler;
import com.d105.loanplease.domain.auth.service.CustomOAuth2UserService;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.exception.CustomAccessDeniedHandler;
import com.d105.loanplease.global.exception.CustomAuthenticationEntryPoint;
import lombok.AllArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.HeadersConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
@Configuration
@EnableWebSecurity
@AllArgsConstructor
@Slf4j
public class SecurityConfig {
    private final CustomOAuth2UserService customOAuth2UserService;
    private final CustomSuccessHandler customSuccessHandler;
    private final TokenProvider tokenProvider;
    private final CustomAccessDeniedHandler customAccessDeniedHandler;
    private final CustomAuthenticationEntryPoint   authenticationEntryPoint;
    @Bean
    public BCryptPasswordEncoder bCryptPasswordEncoder() {

        return new BCryptPasswordEncoder();
    }
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {

        //csrf disable
        //JWT는 세션관리를 stateless로 관리하기 때문에 csrf를 disable로 설정한다.
        http
                .csrf((auth) -> auth.disable());

        // JWT : From 로그인 방식 disable
        http
                .formLogin((auth) -> auth.disable());

        //JWT : HTTP Basic 인증 방식 disable
        http
                .httpBasic((auth) -> auth.disable());

        //순서

        log.info("1");
        //CORS

        log.info("2");
        http
                .headers((headers) -> headers.frameOptions(
                        HeadersConfigurer.FrameOptionsConfig::sameOrigin
                ));


        log.info("3");
        //JWTFilter 추가
        http
                .addFilterBefore(new JWTFilter(tokenProvider), UsernamePasswordAuthenticationFilter.class);
        //7번


        log.info("4");
        //oauth2
        http
                .oauth2Login((oauth2) -> oauth2
                        .userInfoEndpoint((userInfoEndpointConfig) -> userInfoEndpointConfig
                                .userService(customOAuth2UserService))
                        .successHandler(customSuccessHandler)
                );


        log.info("5");
        //경로별 인가 작업
        http
                .authorizeHttpRequests((auth) -> auth
                        .requestMatchers("/").permitAll()
                        .requestMatchers("/api/upload").permitAll()
                        .requestMatchers("/api/server").permitAll()
                                .requestMatchers("/api/auth/nickname/**", "/error").permitAll()
                                .requestMatchers("/swagger-ui/**").permitAll()
                                .requestMatchers("/swagger-resources/**").permitAll()
                                .requestMatchers("/v3/api-docs/**").permitAll()
                                .requestMatchers("/swagger-ui/docs").permitAll()
                                .requestMatchers("/api/auth/register").permitAll()
                        .requestMatchers("/api/auth/refresh").permitAll()
                        .anyRequest().authenticated()
//                                .anyRequest().permitAll()
                ).exceptionHandling(authentication ->        // 7)
                        authentication.authenticationEntryPoint(authenticationEntryPoint) //401일 때
                                .accessDeniedHandler(customAccessDeniedHandler)); //403일 때



        log.info("6");
        //세션 설정 : STATELESS로 설정해야 JWT가 적용이 된다.
        http
                .sessionManagement((session) -> session
                        .sessionCreationPolicy(SessionCreationPolicy.STATELESS));


        log.info("7");
        return http.build();
    }
}