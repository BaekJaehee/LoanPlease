package com.d105.loanplease.domain.user.entity;

import com.d105.loanplease.domain.store.domain.Item;
import com.d105.loanplease.domain.store.domain.Loan;
import io.jsonwebtoken.lang.Assert;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "user_item_tb")
public class UserItem {
    
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_item_id")
    private Long userItemId;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;

    @ManyToOne
    @JoinColumn(name = "item_id")
    private Item item;

    private Integer count;

    public UserItem(Item item, Integer itemCount, User user) {
        this.user = user;
        this.item = item;
        user.getUserItemList().add(this);
        this.count = itemCount;
    }

    // 아이템 상품 구매 (기존 아이템이 없는 경우)
    public static UserItem purchaseItem(Item item, Integer itemCount, User user) {
        UserItem userItem = new UserItem(item, itemCount, user);

        user.getUserItemList().add(userItem);
        user.purchase(item.getPrice()); // 포인트가 부족한지 체크

        return userItem;
    }

    // 아이템 상품 구매 (기존 아이템이 있는 경우)
    public void purchaseItem(Integer price, Integer itemCount, User user) {
        user.purchase(price*itemCount);
        this.count+=itemCount;
    }

    // 아이템 사용
    public void useItem() {
        Assert.isTrue(count>0, "보유한 아이템이 없습니다.");
        this.count--;
    }
}
