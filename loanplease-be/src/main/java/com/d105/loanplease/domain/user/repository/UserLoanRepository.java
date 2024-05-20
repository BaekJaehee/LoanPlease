package com.d105.loanplease.domain.user.repository;

import com.d105.loanplease.domain.user.entity.UserItem;
import com.d105.loanplease.domain.user.entity.UserLoan;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface UserLoanRepository extends JpaRepository<UserLoan, Long> {

    List<UserLoan> findAllByUserUserId(Long userId);
}
