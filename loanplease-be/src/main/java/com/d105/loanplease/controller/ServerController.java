package com.d105.loanplease.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Slf4j
public class ServerController {

    //heath_login
    @GetMapping("/api/hi")
    public String test1() {
        return "hi";
    }
    @GetMapping("/api/server")
    public String healthCheckForServerReload() {
        log.info("server controller test 함수 접속");
        return "secret";
    }

    @GetMapping("/api/test")
    public String apitest() {
        log.info("test 함수 접속");
        return "API hi";
    }

}
