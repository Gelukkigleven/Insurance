from flask import Flask, jsonify
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# 模拟业务流程
def initialize_sale_process():
    # 运营支持部门准备方案或模板
    proposal = "标准家庭财产保险方案"  # 运营支持部门提供

    # 保险代理人、经纪人或客户经理基于方案进行产品讲解
    product_details = {
        "name": "家庭财产保险",
        "coverage": "火灾、盗窃、自然灾害等",
        "price": "1200元/年"
    }

    # 销售人员根据客户情况选择适合的产品并发送保险条款
    available_products = [
        {"name": "基础家庭财产保险", "coverage": "火灾、盗窃", "price": 1000},
        {"name": "高端家庭财产保险", "coverage": "火灾、盗窃、自然灾害", "price": 2000}
    ]

    selected_product = random.choice(available_products)  # 假设随机选择
    terms = selected_product["terms"]  # 假设产品中包含条款

    return {
        "proposal": proposal,
        "product_details": product_details,
        "selected_product": selected_product,
        "terms": terms
    }

# 数据库设计
class FinancialProducts:
    def __init__(self, name, characteristics, risk_level, cost):
        self.name = name
        self.characteristics = characteristics
        self.risk_level = risk_level
        self.cost = cost

# 模拟数据库中的部分金融产品数据
financial_products = [
    FinancialProducts("安心车险", ["全面保障", "理赔快捷"], "低风险", 1500),
    FinancialProducts("家庭财产险", ["多重保障", "定制服务"], "低风险", 1000),
    FinancialProducts("投资型车险", ["收益增长", "保障组合"], "中风险", 2000),
    FinancialProducts("高端财产险", ["专属保障", "尊享服务"], "中风险", 3000),
]

# 模拟产生数据
def generate_sales_data():
    sales_data = []
    for product in financial_products:
        start_date = datetime(2024, 1, 1)
        current_date = start_date
        while current_date < datetime.now():
            # 假设每月有随机的销售金额
            sale_amount = random.uniform(1000, 10000)
            # 保存数据
            sales_data.append({
                "product_id": id(product),
                "product_name": product.name,
                "sale_date": current_date.strftime("%Y-%m-%d"),
                "sale_amount": sale_amount,
                "transaction_id": random.randint(1000, 9999)
            })
            # 模拟每月数据
            current_date += timedelta(days=30)
    return sales_data

# 模拟数据分析
def analyze_sales_data(sales_data):
    # 计算各产品的销售额
    product_sales = {}
    for sale in sales_data:
        product_name = sale["product_name"]
        if product_name not in product_sales:
            product_sales[product_name] = 0
        product_sales[product_name] += sale["sale_amount"]

    # 计算每月的总销售额
    monthly_sales = {}
    for sale in sales_data:
        month = sale["sale_date"][:7]  # 获取年-月
        if month not in monthly_sales:
            monthly_sales[month] = 0
        monthly_sales[month] += sale["sale_amount"]

    return product_sales, monthly_sales

# 模拟行政流程
def administrative_process():
    # 生成保险单并交给后督部门审核
    policy_number = "P20250101-" + str(random.randint(1000, 9999))
    # 假设审核通过
    approval_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"政策编号": policy_number, "批准日期": approval_date}

# 模拟核保流程
def underwriting_process():
    # 审核健康信息等
    # 假设审核通过
    underwriting_result = "已审批"
    underwriting_reason = "客户符合所有标准"
    return {"承保结果": underwriting_result, "原因": underwriting_reason}

# 模拟初审流程
def initial_review():
    # 检查保单状态
    # 假设保单有效
    policy_status = "有效"
    return {"保单状态": policy_status}

# 模拟签单流程
def sign_contract():
    # 签署合同并加盖公章
    contract_number = "C20250101-" + str(random.randint(1000, 9999))
    # 生成业务来源和渠道
    sources = ["直销", "经纪人", "代理人"]
    source = random.choice(sources)
    channels = ["线上平台", "线下门店"]
    channel = random.choice(channels)
    return {
        "合同编号": contract_number,
        "业务来源": source,
        "销售渠道": channel
    }

# 业务数据依托的科目表
business_subjects = {
    "保险产品": ["安心车险", "家庭财产险", "投资型车险", "高端财产险"],
    "销售渠道": ["直销", "经纪人", "代理人", "线上平台", "线下门店"],
    "业务来源": ["公司A", "公司B", "公司C", "其他"]
}

# 模拟业务流程
def business_process():
    # 业务流程的各个步骤
    sale_process = initialize_sale_process()
    admin_process = administrative_process()
    underwriting = underwriting_process()
    initial_review_result = initial_review()
    sign_result = sign_contract()

    return {
        "销售过程": sale_process,
        "行政流程": admin_process,
        "承保流程": underwriting,
        "初审流程": initial_review_result,
        "签单流程": sign_result
    }

# API接口
@app.route('/sales_data', methods=['GET'])
def get_sales_data():
    sales_data = generate_sales_data()
    return jsonify(sales_data)

@app.route('/sales_analysis', methods=['GET'])
def get_sales_analysis():
    sales_data = generate_sales_data()
    product_sales, monthly_sales = analyze_sales_data(sales_data)
    return jsonify({
        "product_sales": product_sales,
        "monthly_sales": monthly_sales
    })

@app.route('/business_process', methods=['GET'])
def get_business_process():
    process = business_process()
    return jsonify(process)

if __name__ == '__main__':
    app.run(debug=True)