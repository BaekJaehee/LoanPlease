package com.d105.loanplease.global.util;

public enum Constant {

    FIRST_SLOT_EXPAND_PRICE(2000),
    SECOND_SLOT_EXPAND_PRICE(5000);

    private final int price;

    Constant(int price) {
        this.price = price;
    }

    public Integer price() {
        return this.price;
    }
}
