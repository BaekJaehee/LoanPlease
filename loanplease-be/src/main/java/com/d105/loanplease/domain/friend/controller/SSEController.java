package com.d105.loanplease.domain.friend.controller;

import com.d105.loanplease.domain.friend.service.NotificationService;
import com.d105.loanplease.domain.friend.service.SseEmitters;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.global.util.SecurityUtil;
import lombok.AllArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import java.io.IOException;
@RestController
@AllArgsConstructor
@RequestMapping("/api")
public class SSEController {
    private final NotificationService notificationService;
    private final SseEmitters sseEmitters;
    private final SecurityUtil securityUtil;
    //처음 연결시 사용된다.
    @GetMapping(value = "/connect", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<SseEmitter> connect() {
        SseEmitter emitter = new SseEmitter(); //에미터 생성
        User user = securityUtil.getCurrentUserDetails();
        Long userId = user.getUserId();
        sseEmitters.add(emitter, userId); // sseEmitter에 해당 유저에 대한 에미터 추가

        try {
            emitter.send(SseEmitter.event() //향후 이벤트가 발생했을 때 해당 클라이언트로 전송하기 위해 저장하고 이써야한다.
                    //처음 연결 시 아무 데이터도 보내지 않으면 재연결 요청시 503error 를 뿜을 수 있다.
                    //처음 연결시 더미 데이터를 넣자
                    .name("connect")
                    .data("connected!"));

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return ResponseEntity.ok(emitter);
    }

}

