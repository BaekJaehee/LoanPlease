package com.d105.loanplease.domain.store.adapter.out;

import com.d105.loanplease.domain.store.domain.Loan;
import org.springframework.data.jpa.repository.JpaRepository;

public interface LoanRepository extends JpaRepository<Loan, Long> {
}
