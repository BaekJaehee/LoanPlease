package com.d105.loanplease.domain.friend.service;

import com.d105.loanplease.domain.friend.dto.FriendDetailDto;
import com.d105.loanplease.domain.friend.dto.response.FriendListRes;
import com.d105.loanplease.domain.friend.repository.FriendRepository;
import com.d105.loanplease.domain.user.entity.Friendship;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.util.BaseResponseBody;
import com.d105.loanplease.global.util.SecurityUtil;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;


@Service
@Slf4j
@AllArgsConstructor
public class FriendService {
    private final SecurityUtil securityUtil;

    private final FriendRepository friendRepository;
    private final UserRepository userRepository;
    private final NotificationService notificationService;

//ResponseEntity.status(HttpStatus.CREATED).body(BaseResponseBody.of("200", tokenResDto));
    @Transactional
    public ResponseEntity<BaseResponseBody<Boolean>> processFriendRequest(boolean accept, Long acceptorId) {

//        User senderNickName = userRepository.findById(requestId);
        User user = securityUtil.getCurrentUserDetails();
        Long userId = user.getUserId(); // 요청에 대한 답변을 한 사람


        return friendRepository.findById(userId).map(request -> { //요청자의 freind 값을 찾는다.
            if (accept) { //수락한 상태 -> 답변자의 테이블에 친구 테이블을 생성한다.
                              // 피답변자의 테이블의 친구 테이블을 생성한다.z
                request.setIsAccess(true);
                notificationService.sendFriendAcceptNotification(request.getFrom().getUserId(), "Your friend request has been accepted by " + request.getTo());

                //양쪽이 저장되어야 한다.
                friendRepository.save(request);

                return ResponseEntity.status(HttpStatus.ACCEPTED).body(BaseResponseBody.of("200",true));
            } else { //거절한 상태
//                friendRequestRepository.delete(request);
                return ResponseEntity.status(HttpStatus.ACCEPTED).body(BaseResponseBody.of("200",true));
            }
        }).orElse(ResponseEntity.status(HttpStatus.BAD_REQUEST).body(BaseResponseBody.of("432",false)));
    }

    @Transactional
    public ResponseEntity<BaseResponseBody<FriendListRes>> getMyFriendList() {
        log.info("Friends");
        User user = securityUtil.getCurrentUserDetails();
        Long userId = user.getUserId();
        log.info("GETUSERINFO"+userId.toString());
        User userDetail = userRepository.getReferenceById(userId);
        Long userId1 = userDetail.getUserId(); // 현재 사용자의 ID를 가져옴
        log.info("userId : "+userId);
        log.info("userId1 : "+userId1);

        List<Friendship> friends = friendRepository.findByFrom_UserId(userId);

        List<Friendship> realFriends = new ArrayList<>();

        for (Friendship value : friends) { //isAccept가 true인 애들만 list에 새로 담는다.
            if (value.getIsAccess()) {
                realFriends.add(value);
                log.info(value.getFrom()+" : "+value.getTo()+" : "+value.getFriendshipId());
            }else{
                log.info(value.getFrom()+" : "+value.getTo()+" : "+value.getFriendshipId());
            }
        }
        List<FriendDetailDto> friendList = realFriends.stream()
                .map(friend -> new FriendDetailDto(friend.getTo().getNickname(), friend.getTo().getEmail(), friend.getTo().getProfileImg(),true))
                .toList();
        FriendListRes friendListRes = FriendListRes.builder().friends(friendList).build();
        return ResponseEntity.ok(BaseResponseBody.of("200", friendListRes));
    }

    @Transactional
    public ResponseEntity<BaseResponseBody<FriendListRes>> searchFriendList(String friendName) {
        List<User> potentialFriends = userRepository.findByNicknameContaining(friendName);
        //먼저 뽑고
        Long userId = securityUtil.getCurrentUserId();
        List<Friendship> friends = friendRepository.findByFrom_UserId(userId);
        List<FriendDetailDto> searchFriendList = new ArrayList<>();

        //돌면서 확인해
        for (User potentialFriend : potentialFriends) {
            Optional<User> findTmpUserInfo = userRepository.findByEmail(potentialFriend.getEmail());
            if (findTmpUserInfo.isPresent()){//나와 친구인지 확인한다.
                User tmpUser = findTmpUserInfo.get(); //검색된 유저들의 정보들을 받는다.
                Long tmpUserId = tmpUser.getUserId(); //잠재 친구의 아이디 값을 가져온다.
                log.info(tmpUserId.toString()+"TOUSERID");
                boolean flag = false;
                for(Friendship friendship : friends){ //검색을 시도한 유저의 친구들의 목록을 가져온다.
                    log.info(friendship.getTo()+" _ My Friend");
                    Long tmpToUser = friendship.getTo().getUserId();
                    if (tmpToUser.equals(tmpUserId)){
                        log.info("WeAreAFriends");
                        searchFriendList.add(new FriendDetailDto(tmpUser.getNickname(), tmpUser.getNickname(), tmpUser.getProfileImg(), friendship.getIsAccess() )); //친구 추가
                        flag = true;
                        break;
                    }
                }
                if (!flag)  searchFriendList.add(new FriendDetailDto(tmpUser.getNickname(), tmpUser.getNickname(), tmpUser.getProfileImg(), false )); //친구 추가
            }
        }
//        log.info(searchFriendList.size().toString);
        FriendListRes friendListRes = FriendListRes.builder().friends(searchFriendList.stream().toList()).build();
        return ResponseEntity.ok(BaseResponseBody.of("200",friendListRes));
    }

}



//List<FriendDetailDto> friendList = potentialFriends.stream()
//        .map(user ->new FriendDetailDto(user.getNickname(), user.getNickname(), user.getProfileImg()))
//        .toList();