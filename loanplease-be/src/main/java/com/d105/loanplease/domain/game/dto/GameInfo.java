package com.d105.loanplease.domain.game.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class GameInfo {
    LoanRequest loan;
    CustomerInfo customerInfo;
    FinancialInfo financialInfo;
    NonFinancialInfo nonFinancialInfo;
    int credit;
}
