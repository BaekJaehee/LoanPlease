package com.d105.loanplease.domain.user.controller;

import com.d105.loanplease.domain.auth.jwt.TokenProvider;
import com.d105.loanplease.domain.user.dto.request.UserSignUpReq;
import com.d105.loanplease.domain.user.dto.response.UserInfoResponse;
import com.d105.loanplease.domain.user.dto.response.UserSignUpRes;
import com.d105.loanplease.domain.user.service.UserService;
import com.d105.loanplease.global.service.RedisService;
import com.d105.loanplease.global.util.BaseResponseBody;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@Validated
@Slf4j
@RequiredArgsConstructor
public class UserController {


    //회원가입
    private final UserService userService;
    private final TokenProvider tokenProvider;
    private final RedisService redisService;


    @PostMapping("/api/auth/register")
    public ResponseEntity<BaseResponseBody> registerUser(
            @RequestBody UserSignUpReq userSignUpReq
    ) throws Exception {
        UserInfoResponse userSignUpRes = userService.signUp(userSignUpReq);
        return ResponseEntity.status(HttpStatus.CREATED).body(BaseResponseBody.of("200", userSignUpRes));
    }
    @GetMapping("/api/auth/logout")
    public ResponseEntity<BaseResponseBody<Boolean>> userLogout(HttpServletRequest request){
        redisService.delValues(request.getHeader("RefreshToken"));
        return ResponseEntity.ok(BaseResponseBody.of("200",true));
    }
    @PutMapping("/api/auth/")
    public ResponseEntity<BaseResponseBody<Void>> updateUser
            (@RequestParam String nickname,
             @RequestParam("profileImage") MultipartFile profileImage) throws IOException {
        userService.updateUserById(nickname, profileImage);
        return ResponseEntity.ok(BaseResponseBody.of("200", null));
    }

    //닉네임이 사용 중인지 확인한다.
    @GetMapping("/api/auth/nickname/{nickname}")
    public ResponseEntity<BaseResponseBody<Boolean>> checkNicknameAvailability(
            @PathVariable("nickname") String nickname) {
        log.info("ASDas");log.info("ASDas");log.info("ASDas");log.info("ASDas");
        boolean isAvailable = userService.isNicknameAvailable(nickname);
        return ResponseEntity.ok(BaseResponseBody.of("200", isAvailable));
    }//end

    @DeleteMapping("/api/auth/") //난중에 구현..
    public ResponseEntity<BaseResponseBody<Void>> deleteUser() {
//        userService.deleteUserById();
        return ResponseEntity.ok(BaseResponseBody.of("200", null));
    }

    @Operation(summary = "유저 정보 조회", description = "유저의 슬롯, 슬롯 수, 보유 아이템, 보유 대출 상품을 조회합니다.")
    @GetMapping("/api/user-info")
    public ResponseEntity<BaseResponseBody<UserInfoResponse>> getUserInfo() {
        UserInfoResponse userInfo = userService.getUserInfo();
        return ResponseEntity.ok(BaseResponseBody.of("200", userInfo));
    }

}




    //신규 유저인지는 구글로부터 정보 받고 거기서 처리한다.
//    @GetMapping("/email/{email}")
//    public ResponseEntity<BaseResponseBody<User>> getUserByEmail(@PathVariable String email) {
//        User user = userService.findUserByEmail(email);
//        return ResponseEntity.ok(BaseResponseBody.of("200", user));
//    }



