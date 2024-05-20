package com.d105.loanplease.domain.friend.repository;

import com.d105.loanplease.domain.user.entity.Friendship;
import com.d105.loanplease.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;

public interface FriendRepository extends JpaRepository<Friendship, Long> {


    //내 친구들의 리스트를 가져온다.
    List<Friendship> findByFrom_UserId(Long userId);
    //FriendShip 엔터티에서 From이라는 값으로 부터 User클래스의 user_id의 값을 가져온다.

    @Query("SELECT f FROM Friendship f WHERE f.from = :user and f.isAccess = :isAccess")
    List<Friendship> findByUserIdAndIsAccepted(User user, Boolean isAccess);


}

