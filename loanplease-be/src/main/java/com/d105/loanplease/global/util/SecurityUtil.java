package com.d105.loanplease.global.util;

import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.exception.ErrorCode;
import com.d105.loanplease.global.exception.Exceptions;
import io.netty.channel.unix.Errors;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

@AllArgsConstructor
@Component
@Slf4j
public class SecurityUtil {

    private final UserRepository userRepository;

    //유저 정보를 가져온다.
    public User getCurrentUserDetails() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        log.info(authentication.toString());
        if (authentication.isAuthenticated()) {
            return userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new Exceptions(ErrorCode.MEMBER_NOT_EXIST)) ;
        }
        throw new IllegalStateException("No authenticated user available");
    }

    public Long getCurrentUserId() {
        User userDetails = getCurrentUserDetails();
        log.info(userDetails.toString());
        if (userDetails != null){
            return userDetails.getUserId();
        }
        throw new IllegalStateException("UserDetails does not contain user ID information");
    }
    public String getCurrentUserEmail(){
        User userDetails = getCurrentUserDetails();
//        if (User instanceof User) {  //
//            return ((User) userDetails).getEmail();
//        }
        if (userDetails != null){
            return userDetails.getEmail();
        }
        throw new IllegalStateException("UserDetails does not contain user ID information");
    }
}
