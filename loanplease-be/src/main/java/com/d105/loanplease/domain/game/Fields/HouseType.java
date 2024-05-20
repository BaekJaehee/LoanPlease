package com.d105.loanplease.domain.game.Fields;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum HouseType {
    MUNICIPAL("Municipal apartment", "시립 아파트"),
    HOUSE("House / apartment", "자가"),
    WITHP("With parents", "부모와 함께 거주"),
    COOP("Co-op apartment", "Co-op 아파트"),
    RENTED("Rented apartment", "임대"),
    OFFICE("Office apartment", "오피스텔");

    private final String englishName;
    private final String koreanName;
}
