package com.d105.loanplease.domain.game.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class NonFinancialInfo {
    String car;
    String reality;
    int childNum;
    String eduType;
    String familyType;
    String houseType;
    int employedDays;
    int familySize;
}
