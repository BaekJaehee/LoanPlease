package com.d105.loanplease.domain.user.repository;

import com.d105.loanplease.domain.user.entity.UserItem;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UserItemRepository extends JpaRepository<UserItem, Long> {

    List<UserItem> findAllByUserUserId(Long userId);
}