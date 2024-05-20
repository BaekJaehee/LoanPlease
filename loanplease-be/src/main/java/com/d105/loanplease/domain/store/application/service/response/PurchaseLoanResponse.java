package com.d105.loanplease.domain.store.application.service.response;

import com.d105.loanplease.domain.user.dto.UserLoanResDto;
import com.d105.loanplease.domain.user.entity.UserLoan;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class PurchaseLoanResponse {

    private Integer remainPoint;
    private List<UserLoanResDto> userLoanResDtoList = new ArrayList<>();

    public PurchaseLoanResponse(Integer remainPoint, List<UserLoan> userLoanList) {
        this.remainPoint = remainPoint;
        for(UserLoan userLoan: userLoanList) {
            userLoanResDtoList.add(new UserLoanResDto(userLoan));
        }
    }
}
