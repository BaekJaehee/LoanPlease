package com.d105.loanplease.domain.store.application.port.in;

import com.d105.loanplease.domain.store.application.service.response.ChooseLoanResponse;
import com.d105.loanplease.domain.store.application.service.response.PurchaseLoanResponse;
import com.d105.loanplease.domain.store.domain.Loan;
import com.d105.loanplease.domain.user.entity.User;
import org.springframework.http.ResponseEntity;

import java.util.List;

public interface LoanUseCase {

    List<Loan> inquiryAllLoans();
    Loan inquiryLoanById(Long id);
    ResponseEntity<PurchaseLoanResponse> purchaseLoan(Long loanId);
    ResponseEntity<ChooseLoanResponse> changeSlot(int slot_1, int slot_2, int slot_3, int slot_4, int slot_5);
}
