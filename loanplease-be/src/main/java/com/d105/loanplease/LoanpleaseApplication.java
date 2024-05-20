package com.d105.loanplease;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@SpringBootApplication
public class LoanpleaseApplication {

	public static void main(String[] args) {
		SpringApplication.run(LoanpleaseApplication.class, args);
	}
	@Bean
	public CorsFilter corsFilter() {
		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
		CorsConfiguration config = new CorsConfiguration();
		config.setAllowCredentials(true);
		config.addAllowedOriginPattern("http://loanplease.kr"); // 허용할 도메인을 명시
		config.addAllowedOriginPattern("https://loanplease.kr");
//		config.addAllowedOriginPattern("http://localhost:8080");// 추가로 허용할 도메인
		config.addAllowedHeader("*");
		config.addAllowedMethod("*");
		source.registerCorsConfiguration("/**", config);
		return new CorsFilter(source);
	}
//	@Bean
//	public CorsFilter corsFilter() {
//		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
//		CorsConfiguration config = new CorsConfiguration();
//		config.setAllowCredentials(true);
//		config.addAllowedOriginPattern("http://loanplease.kr");
//		config.addAllowedOriginPattern("https://loanplease.kr");
//		config.addAllowedOriginPattern("http://localhost:8080");
//		config.addAllowedHeader("*");
//		config.addAllowedMethod("GET");
//		config.addAllowedMethod("POST");
//		config.addAllowedMethod("PUT");
//		config.addAllowedMethod("PATCH");
//		config.addAllowedMethod("DELETE");
//		config.addAllowedMethod("OPTIONS"); // 추가: OPTIONS 메서드 허용
//		source.registerCorsConfiguration("/**", config);
//		return new CorsFilter(source);
//	}



//	@Bean
//	public WebMvcConfigurer corsConfigurer() {
//		return new WebMvcConfigurer() {
//			@Override
//			public void addCorsMappings(CorsRegistry registry) {
//				registry.addMapping("/**")
//						.allowedOriginPatterns("*")
//						.allowedMethods("GET", "POST", "PUT", "DELETE")
//						.allowedHeaders("*")
//						.allowCredentials(true)
//						.maxAge(3000);
//			}
//		};
//	}
}
