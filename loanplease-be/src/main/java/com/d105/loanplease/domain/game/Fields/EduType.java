package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum EduType {
    HIGHER("Higher education", "대학교/대학원 재학"),
    SECONDARY("Secondary / secondary special", "직업 교육"),
    INCOMPLETE("Incomplete higher", "대학 중퇴"),
    LOWER("Lower secondary", "고등학교 졸업"),
    ACADEMIC("Academic degree", "학위 보유");

    private final String englishName;
    private final String koreanName;

}
