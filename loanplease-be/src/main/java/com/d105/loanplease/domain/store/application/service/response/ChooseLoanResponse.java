package com.d105.loanplease.domain.store.application.service.response;

import com.d105.loanplease.domain.user.entity.Slot;
import lombok.Builder;
import lombok.Getter;

@Getter
public class ChooseLoanResponse {

    private Integer slotNum;
    private Integer slot_1;
    private Integer slot_2;
    private Integer slot_3;
    private Integer slot_4;
    private Integer slot_5;

    public ChooseLoanResponse(final Integer slotNum, final Slot slot) {
        this.slotNum = slotNum;
        this.slot_1 = slot.getSlot_1();
        this.slot_2 = slot.getSlot_2();
        this.slot_3 = slot.getSlot_3();
        this.slot_4 = slot.getSlot_4();
        this.slot_5 = slot.getSlot_5();
    }
}
