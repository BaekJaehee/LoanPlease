package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum IncomeType {

    STUDENT("Student", "학생"),
    COMMERCIAL("Commercial associate", "사기업"),
    WORKING("Working", "근로자"),
    STATE("State servant", "공기업");

    // 영문명을 반환하는 메서드
    private final String englishName;
    // 한글명을 반환하는 메서드
    private final String koreanName;

}
