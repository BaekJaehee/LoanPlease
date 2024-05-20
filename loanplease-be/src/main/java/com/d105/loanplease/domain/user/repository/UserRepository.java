package com.d105.loanplease.domain.user.repository;

import com.d105.loanplease.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {

    // 닉네임으로 사용자 찾기 (닉네임 중복 체크용)
    Optional<User> findByNickname(String nickname);


    //친구 찾기기능에서 해당 이름의 문자를 포함한 친구들의 리스트를 찾아준다.
// Updated method to use 'nickname' instead of 'username'
    List<User> findByNicknameContaining(String nickname);

    Optional<User> findByEmail(String email);

    // 사용자 삭제 (회원탈퇴) - ID를 사용
    @Modifying
    @Query("DELETE FROM User u WHERE u.userId = :userId")
    void deleteUserById(Long userId);

    // 사용자 정보 변경 (회원 정보 변경) - ID를 사용
//    @Modifying
//    @Query("UPDATE User u SET u.nickname = :nickname, u.profileImg = :profileImg WHERE u.userId = :userId")
//    void updateUserById(String nickname, String profileImg);
//}
}
