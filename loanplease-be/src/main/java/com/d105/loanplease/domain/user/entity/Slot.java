package com.d105.loanplease.domain.user.entity;


import com.fasterxml.jackson.annotation.JsonManagedReference;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "slot_tb")
@Getter
@Builder
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Slot {

    @Id @GeneratedValue
    @Column(name = "slot_id")
    private Long slotId;

    private int slot_1;
    private int slot_2;
    private int slot_3;
    private int slot_4;
    private int slot_5;

    @OneToOne(mappedBy = "slot")
    @JsonManagedReference
    private User user;

    public static Slot makeSlot(User user) {
        Slot slot = Slot.builder().slot_1(1).slot_2(2).slot_3(3).slot_4(0).slot_5(0).build();
        slot.user = user;
        user.setSlot(slot);
        return slot;
    }

    public void changeSlot(int slot_1, int slot_2, int slot_3, int slot_4, int slot_5) {
        this.slot_1 = slot_1;
        this.slot_2 = slot_2;
        this.slot_3 = slot_3;
        this.slot_4 = slot_4;
        this.slot_5 = slot_5;
    }
}
