package com.d105.loanplease.domain.user.dto;


import com.d105.loanplease.domain.store.domain.Loan;
import com.d105.loanplease.domain.user.entity.UserLoan;
import lombok.Builder;
import lombok.Getter;

@Getter
public class UserLoanResDto {

    private Long userLoanId;
    private Long loanId;
    private String loanName;
    private String content;
    private Double interest;
    private Integer period;
    private Long limitAmount;
    private String color;

    public UserLoanResDto(UserLoan userLoan) {
        this.userLoanId = userLoan.getUserLoanId();
        Loan loan = userLoan.getLoan();
        this.loanId = loan.getLoanId();
        this.loanName = loan.getName();
        this.content = loan.getContent();
        this.interest = loan.getInterest();
        this.period = loan.getPeriod();
        this.limitAmount = loan.getLimitAmount();
        this.color = loan.getColor();
    }
}
