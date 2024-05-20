package com.d105.loanplease.domain.rank.serviceImpl;

import com.d105.loanplease.domain.friend.repository.FriendRepository;
import com.d105.loanplease.domain.rank.dto.Rank;
import com.d105.loanplease.domain.rank.response.RankResponse;
import com.d105.loanplease.domain.rank.service.RankService;
import com.d105.loanplease.domain.user.entity.Friendship;
import com.d105.loanplease.domain.user.entity.User;
import com.d105.loanplease.domain.user.repository.UserRepository;
import com.d105.loanplease.global.util.SecurityUtil;
import jakarta.persistence.EntityManager;
import jakarta.persistence.TypedQuery;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

@Service
public class RankServiceImpl implements RankService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private FriendRepository friendRepository;

    @Autowired
    private SecurityUtil securityUtil;

    @Autowired
    private EntityManager em;

    @Override
    public ResponseEntity<RankResponse> getAllUserRanks() {

        TypedQuery<User> query = em.createQuery("SELECT u FROM User u ORDER BY u.score DESC", User.class);
        query.setMaxResults(10); // 결과를 10개로 제한합니다.
        List<User> results = query.getResultList();
        List<Rank> rankList = makeRankList(results);

        RankResponse response = RankResponse.createRankResponse(HttpStatus.OK.value(), "전체 랭킹을 불러왔습니다.", rankList);

        return ResponseEntity.ok(response);
    }

    @Override
    public ResponseEntity<RankResponse> getFriendsRanks() {

        Long userId = securityUtil.getCurrentUserId(); // 현재 사용자의 ID를 가져옴
        List<Friendship> friends = friendRepository.findByUserIdAndIsAccepted(userRepository.getReferenceById(userId), true);

        List<User> friendUsers = new ArrayList<>();
        for(Friendship friend : friends){
            friendUsers.add(friend.getTo());
        }
        Collections.sort(friendUsers, new Comparator<User>() {
            @Override
            public int compare(User u1, User u2) {
                return u2.getScore() - u1.getScore();
            }
        });

        List<Rank> rankList = makeRankList(friendUsers);

        RankResponse response = RankResponse.createRankResponse(HttpStatus.OK.value(), "친구 랭킹을 불러왔습니다.", rankList);

        return ResponseEntity.ok(response);
    }
    private List<Rank> makeRankList(List<User> users){
        List<Rank> rankList = new ArrayList<>();
        int rank = 1;
        for(User user : users){
            rankList.add(new Rank(rank++, user.getNickname(), user.getScore(), user.getProfileImg()));

            if(rank>10) return rankList;
        }
        return rankList;
    }
}
