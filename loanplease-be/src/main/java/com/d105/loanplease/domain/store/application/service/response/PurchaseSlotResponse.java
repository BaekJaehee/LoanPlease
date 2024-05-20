package com.d105.loanplease.domain.store.application.service.response;

import lombok.Getter;

@Getter
public class PurchaseSlotResponse {

    private Integer remainPoint;
    private Integer slotNum;

    public PurchaseSlotResponse(final Integer remainPoint, final Integer slotNum) {
        this.remainPoint = remainPoint;
        this.slotNum = slotNum;
    }
}
