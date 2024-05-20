package com.d105.loanplease.domain.friend.service;

import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
@AllArgsConstructor
public class NotificationService {

    private final Map<Long, SseEmitter> emitters = new ConcurrentHashMap<>();

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


}
