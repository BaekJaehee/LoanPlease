package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum YN {
    YES("Y", "있음"),
    NO("N", "없음");

    private final String englishName;
    private final String koreanName;
}
