package com.d105.loanplease.domain.game.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class FinancialInfo {
    double incomeTotal;
    String incomeType;
    String occypType;
}
