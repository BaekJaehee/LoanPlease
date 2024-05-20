package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum FamilyType {
    MARRIED("Married", "기혼"),
    CIVIL("Civil marriage","기혼"),
    SEPERATED("Separated", "별거 중"),
    SINGLE("Single / not married", "미혼");


    private final String englishName;
    private final String koreanName;
}
