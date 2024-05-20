package com.d105.loanplease.global.config;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;

import java.util.Arrays;
import java.util.List;

//@OpenAPIDefinition(
//        info=@Info(title = "LoanPlease",
//                description = "대출의 프로세스와 이해를 게임을 통해 경험하는 프로젝트 \n " +
//                        "   싸피 10기 자율 프로젝트 D105팀 ",
//                version ="v3"))
@Configuration
public class SwaggerConfig {

    @Bean
//    public OpenAPI openAPI() {
//        SecurityScheme securityScheme = new SecurityScheme()
//                .type(SecurityScheme.Type.HTTP).scheme("Bearer").bearerFormat("JWT")
//                .in(SecurityScheme.In.HEADER).name("Authorization");
//        SecurityRequirement securityRequirement = new SecurityRequirement().addList("bearerAuth");
//        Server server = new Server();
//        server.setUrl("https://loanplease.kr"); // https://에 접근 가능하게 설정
//
//        return new OpenAPI()
//                .components(new Components().addSecuritySchemes("bearerAuth", securityScheme))
//                .security(Arrays.asList(securityRequirement))
//                .servers(List.of(server));
//    }
    public OpenAPI openAPI() {
//        Info info = new Info()
//                .title("LoanPlease")
//                .version("v0.0.1")
//                .description("""
//                        대출의 프로세스와 이해를 게임을 통해 경험하는 프로젝트
//                        싸피 10기 자율 프로젝트 D105팀 """);

        SecurityScheme bearer = new SecurityScheme()
                .type(SecurityScheme.Type.HTTP)
                .scheme("bearer")
                .bearerFormat("Authorization")
                .in(SecurityScheme.In.HEADER)
                .name(HttpHeaders.AUTHORIZATION);

        // Security 요청 설정
        SecurityRequirement addSecurityItem = new SecurityRequirement();
        addSecurityItem.addList("Authorization");

        Components components = new Components()
                .addSecuritySchemes("Authorization", bearer);

        return new OpenAPI()
                .components(components)
                .addSecurityItem(addSecurityItem)
                .addServersItem(new Server().url("https://loanplease.kr")
                        .description("Default Server URL"))
                .addServersItem(new Server().url("http://localhost:8080")
                        .description("Local Development Server"));
//                .info(info);
    }

}
