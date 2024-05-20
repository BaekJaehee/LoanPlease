package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum GenderType {
    MALE("M", "남성"),
    FEMALE("F","여성");

    private final String englishName;
    private final String koreanName;
}
