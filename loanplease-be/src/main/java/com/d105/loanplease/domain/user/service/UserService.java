package com.d105.loanplease.domain.user.service;

import com.d105.loanplease.domain.auth.jwt.TokenProvider;
import com.d105.loanplease.domain.store.adapter.out.ItemRepository;
import com.d105.loanplease.domain.store.adapter.out.LoanRepository;
import com.d105.loanplease.domain.store.adapter.out.SlotRepository;
import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.store.domain.Loan;
import com.d105.loanplease.domain.user.dto.UserItemResDto;
import com.d105.loanplease.domain.user.dto.UserLoanResDto;
import com.d105.loanplease.domain.user.dto.response.UserInfoResponse;
import com.d105.loanplease.domain.user.entity.Slot;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.entity.UserItem;
import com.d105.loanplease.domain.user.entity.UserLoan;
import com.d105.loanplease.domain.user.repository.UserItemRepository;
import com.d105.loanplease.domain.user.repository.UserLoanRepository;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.domain.user.dto.request.UserSignUpReq;
import com.d105.loanplease.global.exception.ErrorCode;
import com.d105.loanplease.global.exception.Exceptions;
import com.d105.loanplease.global.service.RedisService;
import com.d105.loanplease.global.util.S3Image;
import com.d105.loanplease.global.util.SecurityUtil;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@Slf4j
@RequiredArgsConstructor
public class UserService {


    private final S3Image imageSave; //S3 서비스 저장
    private final Logger logger = LoggerFactory.getLogger(UserService.class);
    private final SecurityUtil securityUtil;
    private final TokenProvider tokenProvider;
    private final HttpServletResponse response;
    private final UserRepository userRepository;
    private final SlotRepository slotRepository;
    private final LoanRepository loanRepository;
    private final ItemRepository itemRepository;
    private final RedisService redisService;

    private final UserItemRepository userItemRepository;
    private final UserLoanRepository userLoanRepository;

    @Value("${spring.jwt.access.header}")
    private String accessHeader;
    @Value("${spring.jwt.refresh.header}")
    private String refreshHeader;

    public User findUserByEmail(String email) {
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new UsernameNotFoundException("User not found with email: " + email));
    }

    private String saveImage(MultipartFile image, String userEmail) throws IOException {
        String mainImgUrl = "";
        if (image!=null){
            //널이 아니면 저장한다.
            mainImgUrl = imageSave.saveImageS3(image,"updated"+ userEmail + ".png", "/userProfileImage");
        }
        return mainImgUrl;
    }

    //회원 가입 기능
//    UserInfoResponse
    @Transactional
    public UserInfoResponse signUp(UserSignUpReq userReq) throws IOException {

        if (userRepository.findByEmail(userReq.getEmail()).isPresent()){
            //기존에 회원이 존재한다면?
            throw new Exceptions(ErrorCode.EMAIL_EXIST);
        }

        User newUser = User.builder()
                .nickname(userReq.getNickname())
                .email(userReq.getEmail())
                .profileImg(userReq.getProfileImage())
                .score(0)
                .slotNum(3)
                .point(0)
                .role(" ")
                .userItemList(new ArrayList<>())
                .userLoanList(new ArrayList<>())
                .build();
        Slot slot = Slot.makeSlot(newUser);//Slot.makeSlot();

        slotRepository.save(slot);
        userRepository.save(newUser);
        initLoanSetting(newUser);
        initItemSetting(newUser);

        List<UserItemResDto> userItemResDtoList = new ArrayList<>();
        List<UserLoanResDto> userLoanResDtoList = new ArrayList<>();

        List<UserItem> userItemList = newUser.getUserItemList();
        List<UserLoan> userLoanList = newUser.getUserLoanList();

        for(UserItem userItem: userItemList) {
            userItemResDtoList.add(new UserItemResDto(userItem));
        }

        for(UserLoan userLoan: userLoanList) {
            userLoanResDtoList.add(new UserLoanResDto(userLoan));
        }

        //엑세스 토큰을 준다.
        String accessToken = tokenProvider.createAccessJwt(userReq.getEmail());
        String refreshToken = tokenProvider.createRefreshJwt(accessToken);
        redisService.setValues(refreshToken, userReq.getEmail()); //refresh로 저장한다.
        //토큰을 redis에 올린다.
//        tokenProvider.updateTokenRepo(newUser.getEmail(), accessToken);
        Optional<String> tmpString= tokenProvider.extractEmail(refreshToken);
        log.info("REDIS:RefreshToken : " + tmpString.toString());
        //토큰을 쿠키로 준다.
        response.addCookie(createCookie("Authorization", accessToken));
        //return refresh Token
        response.addCookie(createHttpOnlyCookie("RefreshToken",refreshToken));



        return UserInfoResponse.builder().nickname(newUser.getNickname()).profileImage(newUser.getProfileImg()).
                email(newUser.getEmail())
                .point(newUser.getPoint())
                .slot_1(slot.getSlot_1())
                .slot_2(slot.getSlot_2())
                .slot_3(slot.getSlot_3())
                .slot_4(slot.getSlot_4())
                .slot_5(slot.getSlot_5())
                .userItemList(userItemResDtoList)
                .userLoanList(userLoanResDtoList)
                .slotNum(3).build();
//        return UserSignUpRes.builder()
//                .nickname(newUser.getNickname())
//                .email(newUser.getEmail())
//                .profileImg(newUser.getProfileImg())
//                .score(newUser.getScore())
//                .slotNum(newUser.getSlotNum())
//                .build();

    }

    @Transactional
    public void initLoanSetting(User user) {
        Loan loan1 = loanRepository.findById(1L)
                .orElseThrow(() -> new IllegalArgumentException("1번 대출 상품 어디??"));

        Loan loan2 = loanRepository.findById(2L)
                .orElseThrow(() -> new IllegalArgumentException("2번 대출 상품 어디??"));

        Loan loan3 = loanRepository.findById(3L)
                .orElseThrow(() -> new IllegalArgumentException("3번 대출 상품 어디??"));

        UserLoan userLoan1 = new UserLoan(loan1, user);
        UserLoan userLoan2 = new UserLoan(loan2, user);
        UserLoan userLoan3 = new UserLoan(loan3, user);

        userLoanRepository.save(userLoan1);
        userLoanRepository.save(userLoan2);
        userLoanRepository.save(userLoan3);
    }

    @Transactional
    public void initItemSetting(User user) {
        List<Item> itemList = itemRepository.findAll();

        for(Item item: itemList) {
            UserItem userItem = new UserItem(item, 0, user);
            userItemRepository.save(userItem);
        }
    }


    // 닉네임 중복 체크 기능
    public boolean isNicknameAvailable(String nickname) {
        return userRepository.findByNickname(nickname).isEmpty();
    }

    private Cookie createCookie(String key, String value) {
        Cookie cookie = new Cookie(key, value);
        cookie.setMaxAge(60*60*60);
        //cookie.setSecure(true);
        cookie.setPath("/");
        cookie.setHttpOnly(true);
        return cookie;
    }
    private Cookie createHttpOnlyCookie(String key, String value) {
        Cookie cookie = new Cookie(key, value);
        cookie.setMaxAge(604800);  // 여기서는 리프레시 토큰의 유효 기간을 설정
        cookie.setPath("/");
        cookie.setHttpOnly(true);  // JS를 통한 접근 방지
        return cookie;
    }


    // 회원 탈퇴 기능 -> 나중 기능으로 추가
    @Transactional
    public void deleteUserById(Long userId) {
        userRepository.deleteUserById(userId);
    }


    // 회원 정보 변경 기능
    @Transactional
    public void updateUserById( String nickname, MultipartFile image) throws IOException {
        try{
            //JWT를 통해 회원 아이디값 가져오기
//            String userEmail = SecurityUtil.getCurrentUserEmail();
            User getUserInfo = securityUtil.getCurrentUserDetails();
            String userEmail = getUserInfo.getEmail();
            Long userId = getUserInfo.getUserId();
            //S3에 사진 저장
            String mainImgUrl = saveImage(image,userEmail);
            //빌더
            // 기존 User 정보에 새 데이터 적용
            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new IllegalStateException("User not found with ID: " + userId));
            user.setNickname(nickname);
            user.setProfileImg(mainImgUrl);
            //save
            // 변경된 사용자 정보 저장
            userRepository.save(user);
        } catch (IllegalStateException e) {
            // 예외를 호출자에게 전달
            throw new IllegalStateException("Failed to update user: " + e.getMessage(), e);
        }
    }

    public UserInfoResponse getUserInfo() {
        User user = securityUtil.getCurrentUserDetails();
        Long userId = user.getUserId();
        log.info("GETUSERINFO"+userId.toString());
        User userDetail = userRepository.getReferenceById(userId);
        log.info(userDetail.toString());
        Integer slotNum = userDetail.getSlotNum();
        Slot slot = userDetail.getSlot();
        String nickname = userDetail.getNickname();
        String email = userDetail.getEmail();
        String profileImage = userDetail.getProfileImg();
        Integer point = userDetail.getPoint();


        List<UserItem> userItemList = userItemRepository.findAllByUserUserId(userId);
        List<UserLoan> userLoanList = userLoanRepository.findAllByUserUserId(userId);

        List<UserItemResDto> userItemDto = new ArrayList<>();
        List<UserLoanResDto> userLoanDto = new ArrayList<>();

        for(UserItem token : userItemList){
            userItemDto.add(new UserItemResDto(token));
        }
        for (UserLoan token : userLoanList) {
            userLoanDto.add(new UserLoanResDto(token));
        }
        return UserInfoResponse.builder().nickname(nickname).profileImage(profileImage).
                 email(email)
                .slotNum(slotNum)
                .slot_1(slot.getSlot_1())
                .slot_2(slot.getSlot_2())
                .slot_3(slot.getSlot_3())
                .slot_4(slot.getSlot_4())
                .slot_5(slot.getSlot_5())
                .point(point)
                .userItemList(userItemDto)
                .userLoanList(userLoanDto)
                .slotNum(slotNum).build();
    }
}
