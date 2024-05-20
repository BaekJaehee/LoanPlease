package com.d105.loanplease.domain.auth.oauth;

import com.d105.loanplease.domain.user.entity.User;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.oauth2.core.user.OAuth2User;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

public class CustomOAuth2User implements OAuth2User {

    private final User userDTO;

    public CustomOAuth2User(User userDTO) {
        this.userDTO = userDTO;
    }

    @Override
    public Map<String, Object> getAttributes() {
        return null;
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        Collection<GrantedAuthority> collection = new ArrayList<>();
        collection.add(new GrantedAuthority() {
            @Override
            public String getAuthority() {

                return userDTO.getRole();
            }
        });

        return collection;
    }
    //spring 시큐리티에서 중요하다.
    //여기 getName이 중요!!!
    //식별자의 역할
    @Override
    public String getName() {
        return userDTO.getEmail();
    }
    public String getPicture(){
        return userDTO.getProfileImg();
    }
    public String getRole(){
        return userDTO.getRole();
    }
}