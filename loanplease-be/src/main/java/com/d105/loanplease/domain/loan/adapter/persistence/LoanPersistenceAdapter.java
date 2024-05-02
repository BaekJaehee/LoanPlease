package com.d105.loanplease.domain.loan.adapter.persistence;

import com.d105.loanplease.domain.loan.adapter.out.LoanRepository;
import com.d105.loanplease.domain.loan.application.port.out.LoanPort;
import com.d105.loanplease.domain.loan.domain.Loan;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class LoanPersistenceAdapter implements LoanPort {

    @Autowired
    private LoanRepository loanRepository;

    @Override
    public List<Loan> findAll() {
        return loanRepository.findAll();
    }

    @Override
    public Loan findById(final Long id) {
        return loanRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("대출 상품이 존재하지 않습니다."));
    }
}
