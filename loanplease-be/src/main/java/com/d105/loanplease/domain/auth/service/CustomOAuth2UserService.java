package com.d105.loanplease.domain.auth.service;

import com.d105.loanplease.domain.auth.oauth.CustomOAuth2User;
import com.d105.loanplease.domain.auth.oauth.GoogleResponse;
import com.d105.loanplease.domain.auth.oauth.OAuth2Response;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.repository.UserRepository;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
@Slf4j
public class CustomOAuth2UserService extends DefaultOAuth2UserService {
    private final UserRepository userRepository;

    //구글로부터 사용자 정보를 가져오고 내부 시스템의 User 엔터티로 변환하고 이를 CustomOAuth2User 인스턴스로 감싸서 반환하는 역할

    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {

        OAuth2User oAuth2User = super.loadUser(userRequest);
        //유저 정보
        System.out.println(oAuth2User);

        String registrationId = userRequest.getClientRegistration().getRegistrationId();
        OAuth2Response oAuth2Response = null;
        if (registrationId.equals("google")) {

            //한번 더 감싸준다. GoogleResponse 클래스로 한 클래스를 받는다.
            oAuth2Response = new GoogleResponse(oAuth2User.getAttributes());
            log.info("구글로그인 받아옴 : {}", oAuth2User.getAttributes());

        }
        else {
            return null;
        }
        User userDTO = User.builder().email(oAuth2Response.getEmail()).profileImg(oAuth2Response.getPicture()).role("ROLE_USER").build();

        return new CustomOAuth2User(userDTO);


    }
}