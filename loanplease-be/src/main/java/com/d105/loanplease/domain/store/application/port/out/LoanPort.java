package com.d105.loanplease.domain.store.application.port.out;

import com.d105.loanplease.domain.store.domain.Loan;

import java.util.List;

public interface LoanPort {

    List<Loan> findAll();
    Loan findById(Long id);
}
