package com.d105.loanplease.domain.store.application.service.response;

import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.store.domain.Loan;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

@Getter
public class InquiryStoreResponse {

    List<Item> itemList = new ArrayList<>();
    List<Loan> loanList = new ArrayList<>();

    public InquiryStoreResponse() {}

    public void addItem(Item item) {
        itemList.add(item);
    }

    public void addLoan(Loan loan) {
        loanList.add(loan);
    }
}