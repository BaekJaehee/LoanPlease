package com.d105.loanplease.domain.friend.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
@Slf4j
public class SseEmitters {  

    //Tread-safe한 자료구조 ConpyOnWirteArrayList
//    private final List<SseEmitter> emitters = new CopyOnWriteArrayList<>();
    private final Map<Long, SseEmitter> emitters = new ConcurrentHashMap<>();
    public SseEmitter add(SseEmitter emitter, Long id) {
        //특정 id 에 따른 emiiter를 저장한다.
        this.emitters.put(id,emitter);
        log.info("new emitter added: {}", emitter);
        log.info("emitter list size: {}", emitters.size());
        emitter.onCompletion(() -> {  
            log.info("onCompletion callback");
            //성공하면 기존에 있는 토큰을 삭제한다.
            this.emitters.remove(emitter);    // 만료되면 리스트에서 삭제
        });  
        emitter.onTimeout(() -> {  
            log.info("onTimeout callback");  
            emitter.complete();  
        });
        return emitter;  
    }

    public void sendFriendRequestNotification(Long receiverId, String message) { //친구 요청을 보낸다.
        SseEmitter emitter = emitters.get(receiverId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event().name("FRIEND_REQUEST").data(message));
            } catch (IOException e) {
                emitter.completeWithError(e);
            }
        }
    }

    public void sendFriendDenyNotification(Long userId, String message) { ///친구 요청을 거절한다.
        SseEmitter emitter = emitters.get(userId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event().name("FRIEND_DENY").data(message));
            } catch (IOException e) {
                emitter.completeWithError(e);
            }
        }
    }

    public void sendFriendAcceptNotification(Long userId, String message) { ///친구 요청을 승인한다.
        SseEmitter emitter = emitters.get(userId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event().name("FRIEND_ACCEPT").data(message));
            } catch (IOException e) {
                emitter.completeWithError(e);
            }
        }
    }

}